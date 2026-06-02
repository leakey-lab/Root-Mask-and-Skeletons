"""
UI Panel creation methods for the main window.
Handles left panel (controls/tree) and right panel (display area) creation.
"""

import os
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QLineEdit,
    QLabel,
    QComboBox,
    QFrame,
    QSizePolicy,
    QStackedWidget,
)
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon

from app.gui.widgets import tokens
from app.gui.widgets.icons import load_icon


def get_icon_path(icon_name: str) -> str:
    """
    Get the full path to an icon file.
    
    Args:
        icon_name: Name of the icon file (with extension)
        
    Returns:
        Full path to the icon file
    """
    # Check in resources/icons first
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    resources_icons_path = os.path.join(base_path, "resources", "icons", icon_name)
    if os.path.exists(resources_icons_path):
        return resources_icons_path
    
    # Check in app/utils for XML icons
    app_utils_path = os.path.join(base_path, "app", "utils", icon_name)
    if os.path.exists(app_utils_path):
        return app_utils_path
    
    return ""


def create_icon_button(icon_path: str, tooltip: str, size: int = 100) -> QToolButton:
    """
    Create a QToolButton with an icon and tooltip.
    
    Args:
        icon_path: Path to the icon file (SVG or XML)
        tooltip: Tooltip text for the button
        size: Button size in pixels
        
    Returns:
        Configured QToolButton
    """
    button = QToolButton()
    button.setToolTip(tooltip)
    button.setFixedSize(QSize(size, size))
    # Use full button size for icon
    button.setIconSize(QSize(size, size))
    
    if icon_path and os.path.exists(icon_path):
        button.setIcon(QIcon(icon_path))
    
    return button


def create_text_button(text: str, tooltip: str) -> QPushButton:
    """
    Create a QPushButton with text and tooltip.
    
    Args:
        text: Button text label
        tooltip: Tooltip text for the button
        
    Returns:
        Configured QPushButton
    """
    button = QPushButton(text)
    button.setToolTip(tooltip)
    return button


def create_section_label(text: str, color: str = "#bd93f9") -> QLabel:
    """
    Create a styled section label for button groups.
    
    Args:
        text: Label text
        color: Color for the label text (hex color code)
        
    Returns:
        Styled QLabel
    """
    label = QLabel(text)
    label.setStyleSheet(f"""
        QLabel {{
            color: {color};
            font-size: 8pt;
            font-weight: bold;
            padding: 2px 0px;
            margin-top: 4px;
        }}
    """)
    return label


def create_left_panel(main_window) -> QWidget:
    """
    Create the left panel with toolbar buttons, view mode selector, and file tree.
    
    Args:
        main_window: MainWindow instance to connect signals and store widgets
        
    Returns:
        QWidget: The configured left panel widget
    """
    left_widget = QWidget()
    layout = QVBoxLayout(left_widget)
    layout.setSpacing(6)
    layout.setContentsMargins(8, 8, 8, 8)

    # Style for text toolbar buttons
    text_button_style = """
        QPushButton {
            background-color: #44475a;
            color: #f8f8f2;
            border: 2px solid #6272a4;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 9pt;
            font-weight: bold;
            text-align: center;
        }
        QPushButton:hover {
            background-color: #6272a4;
            border: 2px solid #8be9fd;
        }
        QPushButton:pressed {
            background-color: #5262a4;
            border: 2px solid #bd93f9;
        }
    """

    # Style for full-width primary button (Load Images)
    primary_button_style = """
        QPushButton {
            background-color: #6272a4;
            color: #f8f8f2;
            border: none;
            border-radius: 6px;
            padding: 10px 16px;
            font-weight: bold;
            font-size: 10pt;
        }
        QPushButton:hover {
            background-color: #7282b4;
            border: 2px solid #bd93f9;
        }
        QPushButton:pressed {
            background-color: #5262a4;
        }
    """

    # ========== LOAD IMAGES BUTTON (Full width, prominent) ==========
    load_icon_path = get_icon_path("folder-open.svg")
    main_window.load_images_button = QPushButton("  Load Images")
    if load_icon_path:
        main_window.load_images_button.setIcon(QIcon(load_icon_path))
        main_window.load_images_button.setIconSize(QSize(24, 24))
    main_window.load_images_button.setStyleSheet(primary_button_style)
    main_window.load_images_button.setToolTip("Load images from a directory")
    main_window.load_images_button.clicked.connect(main_window.load_images)
    # FIX1: Welcome screen is the sole load entry. The attr/connect stay alive
    # for any code referencing them, but the button is not shown in the shell.
    main_window.load_images_button.setVisible(False)

    # ====================================================================== #
    # FIX4: The stage action buttons are constructed here (so their mw.<attr>
    # names and .clicked.connect() targets stay intact) but are NOT added to
    # the left layout. build_action_bar() reparents the real button objects
    # into the stage-aware action bar. Construction-only here.
    # ====================================================================== #

    # Generate ML Masks button (Mask stage)
    main_window.generate_mask_button = create_text_button(
        "Generate ML Masks",
        "Generate ML Masks\n\nUse AI machine learning to automatically generate segmentation masks for root images. This will process all loaded images using a trained neural network model."
    )
    main_window.generate_mask_button.setStyleSheet(text_button_style)
    main_window.generate_mask_button.clicked.connect(
        main_window.mask_generation_handler.generate_masks
    )

    # Generate Skeleton button (Skeleton stage)
    main_window.generate_button = create_text_button(
        "Generate Skeleton",
        "Generate Skeleton\n\nExtract skeleton structure (medial axis) from segmentation masks. Converts mask images into line-based skeleton representations for length measurement."
    )
    main_window.generate_button.setStyleSheet(text_button_style)
    main_window.generate_button.clicked.connect(main_window.skeleton_handler.generate_skeleton)

    # Skeleton Correction button (reparented; kept for .setText callers)
    main_window.toggle_skeleton_correction_button = create_text_button(
        "Skeleton Correction",
        "Skeleton Correction\n\nOpen the skeleton correction editor to manually fix skeleton errors. Allows you to add, remove, or modify skeleton branches interactively."
    )
    main_window.toggle_skeleton_correction_button.setStyleSheet(text_button_style)
    main_window.toggle_skeleton_correction_button.clicked.connect(
        main_window.toggle_skeleton_correction
    )

    # Toggle Mask Tracing button (reparented; kept for .setText callers)
    main_window.toggle_mask_tracing_button = create_text_button(
        "Mask Tracing",
        "Toggle Mask Tracing\n\nSwitch to mask tracing mode to manually draw or edit segmentation masks using brush tools. Toggle again to return to the main view."
    )
    main_window.toggle_mask_tracing_button.setStyleSheet(text_button_style)
    main_window.toggle_mask_tracing_button.clicked.connect(main_window.toggle_mask_tracing)

    # Calculate Root Length button (Measure stage)
    main_window.calculate_length_button = create_text_button(
        "Calculate Root Length",
        "Calculate Root Length\n\nMeasure total root length from skeleton data. Processes all skeleton files and calculates cumulative length measurements for each image."
    )
    main_window.calculate_length_button.setStyleSheet(text_button_style)
    main_window.calculate_length_button.clicked.connect(main_window.calculate_root_length)

    # Calculate Root Area button (Measure stage)
    main_window.calculate_area_button = create_text_button(
        "Calculate Root Area",
        "Calculate Root Area\n\nMeasure root surface area from segmentation masks. Processes all mask files and calculates pixel-based area measurements for each image."
    )
    main_window.calculate_area_button.setStyleSheet(text_button_style)
    main_window.calculate_area_button.clicked.connect(main_window.calculate_root_area)

    # Visualize Root Length button (kept for .setText callers; Visualize stage
    # uses the SegmentedControl in the action bar)
    main_window.visualize_root_length_button = create_text_button(
        "Visualize Root Length",
        "Visualize Root Length\n\nOpen interactive dashboard with line charts showing root length analysis. Compare length measurements across different images, tubes, and dates."
    )
    main_window.visualize_root_length_button.setStyleSheet(text_button_style)
    main_window.visualize_root_length_button.clicked.connect(
        main_window.toggle_root_length_visualization
    )

    # Visualize Root Area button (kept for .setText callers)
    main_window.visualize_root_area_button = create_text_button(
        "Visualize Root Area",
        "Visualize Root Area\n\nOpen interactive dashboard with bar charts showing root area analysis. Compare area measurements across different images, tubes, and dates."
    )
    main_window.visualize_root_area_button.setStyleSheet(text_button_style)
    main_window.visualize_root_area_button.clicked.connect(
        main_window.toggle_root_area_visualization
    )

    # ========== VIEW MODE COMBO (hidden live driver) ==========
    # The combo stays the single source of truth for the display mode (indices
    # 0/1/2 -> display_controller.update_display_mode). It is hidden; a cosmetic
    # SegmentedControl in the action bar forwards into it (see build_action_bar).
    main_window.view_mode_combo = QComboBox()
    main_window.view_mode_combo.addItems(["Single Image", "Overlay", "Side by Side"])
    main_window.view_mode_combo.currentIndexChanged.connect(
        main_window.display_controller.update_display_mode
    )
    main_window.view_mode_combo.setVisible(False)
    layout.addWidget(main_window.view_mode_combo)

    # ========== SEARCH FIELD ==========
    main_window.library_search = QLineEdit()
    main_window.library_search.setPlaceholderText("Search images…")
    main_window.library_search.setClearButtonEnabled(True)
    main_window.library_search.setStyleSheet(f"""
        QLineEdit {{
            background-color: {tokens.BG_3};
            color: {tokens.TEXT};
            border: 1px solid {tokens.BORDER};
            border-radius: 7px;
            padding: 6px 10px;
            font-size: 12px;
        }}
        QLineEdit:focus {{ border: 1px solid {tokens.ACCENT_LINE}; }}
    """)
    main_window.library_search.textChanged.connect(
        lambda text: filter_file_list(main_window, text)
    )
    layout.addWidget(main_window.library_search)

    # ========== TREE CONTROL BUTTONS (small icon buttons) ==========
    tree_controls_layout = QHBoxLayout()
    tree_controls_layout.setSpacing(5)

    tree_button_style = f"""
        QPushButton {{
            background-color: {tokens.BG_2};
            color: {tokens.TEXT_MUTED};
            border: 1px solid {tokens.BORDER};
            border-radius: 6px;
            padding: 5px 10px;
            font-size: 8pt;
        }}
        QPushButton:hover {{
            background-color: {tokens.BG_3};
            color: {tokens.TEXT};
            border: 1px solid {tokens.BORDER_STRONG};
        }}
    """

    main_window.expand_all_button = QPushButton(" Expand")
    main_window.expand_all_button.setIcon(load_icon("expand", tokens.TEXT_MUTED, 14))
    main_window.expand_all_button.setStyleSheet(tree_button_style)
    main_window.expand_all_button.clicked.connect(lambda: main_window.file_list.expandAll())
    main_window.expand_all_button.setMaximumHeight(28)
    tree_controls_layout.addWidget(main_window.expand_all_button)

    main_window.collapse_all_button = QPushButton(" Collapse")
    main_window.collapse_all_button.setIcon(load_icon("collapse", tokens.TEXT_MUTED, 14))
    main_window.collapse_all_button.setStyleSheet(tree_button_style)
    main_window.collapse_all_button.clicked.connect(lambda: main_window.file_list.collapseAll())
    main_window.collapse_all_button.setMaximumHeight(28)
    tree_controls_layout.addWidget(main_window.collapse_all_button)

    layout.addLayout(tree_controls_layout)

    # ========== IMAGE LIBRARY LABEL ==========
    images_label = QLabel("Image Library")
    images_label.setStyleSheet("""
        QLabel {
            color: #f8f8f2;
            font-size: 10pt;
            font-weight: bold;
            padding: 4px 0px;
        }
    """)
    layout.addWidget(images_label)

    # ========== FILE TREE ==========
    main_window.file_list = QTreeWidget()
    main_window.file_list.setHeaderLabels(["Field → Tube → Date → Images"])
    main_window.file_list.itemClicked.connect(main_window.on_tree_item_clicked)
    main_window.file_list.setAlternatingRowColors(True)
    main_window.file_list.setAnimated(True)
    main_window.file_list.setIndentation(20)

    # Enhanced styling for the tree widget
    main_window.file_list.setStyleSheet("""
        QTreeWidget {
            background-color: #282a36;
            color: #f8f8f2;
            border: 2px solid #44475a;
            border-radius: 5px;
            padding: 5px;
            font-size: 9pt;
        }
        QTreeWidget::item {
            padding: 5px;
            border-radius: 3px;
        }
        QTreeWidget::item:hover {
            background-color: #44475a;
        }
        QTreeWidget::item:selected {
            background-color: #6272a4;
            color: #f8f8f2;
        }
        QTreeWidget::branch {
            background-color: #282a36;
        }
        QTreeWidget::branch:has-children:!has-siblings:closed,
        QTreeWidget::branch:closed:has-children:has-siblings {
            image: url(none);
            border-image: none;
        }
        QTreeWidget::branch:open:has-children:!has-siblings,
        QTreeWidget::branch:open:has-children:has-siblings {
            image: url(none);
            border-image: none;
        }
    """)

    layout.addWidget(main_window.file_list, 1)

    return left_widget


def filter_file_list(main_window, query: str) -> None:
    """Filter the image-library tree by item text (case-insensitive contains).

    FIX5: empty query shows everything. A parent stays visible if any of its
    descendants match (so the path to a matching image is preserved).
    """
    tree = getattr(main_window, "file_list", None)
    if tree is None:
        return
    needle = (query or "").strip().lower()

    def visit(item: "QTreeWidgetItem") -> bool:
        child_match = False
        for i in range(item.childCount()):
            child_match = visit(item.child(i)) or child_match
        self_match = needle in item.text(0).lower() if needle else True
        visible = self_match or child_match
        item.setHidden(not visible)
        return visible

    for i in range(tree.topLevelItemCount()):
        visit(tree.topLevelItem(i))


def create_right_panel(main_window) -> QWidget:
    """
    Create the right panel with the display area.
    
    Args:
        main_window: MainWindow instance to connect display controller
        
    Returns:
        QWidget: The configured right panel widget
    """
    from .empty_state import EmptyStateWidget

    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)

    # Page stack: index 0 = real display area, index 1 = empty-state placeholder.
    page_stack = QStackedWidget()

    display_page = QWidget()
    main_window.display_area = main_window.display_controller.setup_display_area(
        display_page
    )
    page_stack.addWidget(display_page)

    empty_state = EmptyStateWidget(on_load=main_window.load_images)
    page_stack.addWidget(empty_state)

    # Start on the empty state until images are loaded; MainWindow flips to the
    # display page in on_loading_finished.
    page_stack.setCurrentIndex(1)

    main_window.display_page_stack = page_stack
    main_window.empty_state = empty_state

    outer.addWidget(page_stack)
    return container
