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
    QLabel,
    QComboBox,
    QFrame,
    QSizePolicy,
    QStackedWidget,
)
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon


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

    # Style for icon toolbar buttons
    toolbar_button_style = """
        QToolButton {
            background-color: transparent;
            border: 2px solid transparent;
            border-radius: 6px;
            padding: 4px;
        }
        QToolButton:hover {
            background-color: #44475a;
            border: 2px solid #6272a4;
        }
        QToolButton:pressed {
            background-color: #6272a4;
            border: 2px solid #bd93f9;
        }
    """
    
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

    # ========== TOP ROW: PROCESSING & CORRECTION/ANNOTATIONS ==========
    top_row_layout = QHBoxLayout()
    top_row_layout.setSpacing(8)
    top_row_layout.setContentsMargins(0, 0, 0, 0)

    # Left column: Processing section
    processing_widget = QWidget()
    processing_widget.setStyleSheet("""
        QWidget {
            background-color: rgba(139, 233, 253, 0.1);
            border: 2px solid #8be9fd;
            border-radius: 6px;
            padding: 6px;
        }
    """)
    processing_section = QVBoxLayout(processing_widget)
    processing_section.setSpacing(4)
    processing_section.setContentsMargins(0, 0, 0, 0)
    processing_section.addWidget(create_section_label("Processing", "#8be9fd"))
    
    processing_layout = QVBoxLayout()
    processing_layout.setSpacing(4)
    processing_layout.setContentsMargins(0, 0, 0, 0)

    # Generate ML Masks button
    main_window.generate_mask_button = create_text_button(
        "Generate ML Masks",
        "Generate ML Masks\n\nUse AI machine learning to automatically generate segmentation masks for root images. This will process all loaded images using a trained neural network model."
    )
    main_window.generate_mask_button.setStyleSheet(text_button_style)
    main_window.generate_mask_button.clicked.connect(
        main_window.mask_generation_handler.generate_masks
    )
    processing_layout.addWidget(main_window.generate_mask_button)

    # Generate Skeleton button
    main_window.generate_button = create_text_button(
        "Generate Skeleton",
        "Generate Skeleton\n\nExtract skeleton structure (medial axis) from segmentation masks. Converts mask images into line-based skeleton representations for length measurement."
    )
    main_window.generate_button.setStyleSheet(text_button_style)
    main_window.generate_button.clicked.connect(main_window.skeleton_handler.generate_skeleton)
    processing_layout.addWidget(main_window.generate_button)

    processing_section.addLayout(processing_layout)
    processing_widget.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
    )
    top_row_layout.addWidget(processing_widget, 1)

    # Right column: Correction & Annotations section
    correction_widget = QWidget()
    correction_widget.setStyleSheet("""
        QWidget {
            background-color: rgba(255, 184, 108, 0.1);
            border: 2px solid #ffb86c;
            border-radius: 6px;
            padding: 6px;
        }
    """)
    correction_section = QVBoxLayout(correction_widget)
    correction_section.setSpacing(4)
    correction_section.setContentsMargins(0, 0, 0, 0)
    correction_section.addWidget(create_section_label("Correction & Annotations", "#ffb86c"))
    
    correction_layout = QVBoxLayout()
    correction_layout.setSpacing(4)
    correction_layout.setContentsMargins(0, 0, 0, 0)

    # Skeleton Correction button
    main_window.toggle_skeleton_correction_button = create_text_button(
        "Skeleton Correction",
        "Skeleton Correction\n\nOpen the skeleton correction editor to manually fix skeleton errors. Allows you to add, remove, or modify skeleton branches interactively."
    )
    main_window.toggle_skeleton_correction_button.setStyleSheet(text_button_style)
    main_window.toggle_skeleton_correction_button.clicked.connect(
        main_window.toggle_skeleton_correction
    )
    correction_layout.addWidget(main_window.toggle_skeleton_correction_button)

    # Toggle Mask Tracing button
    main_window.toggle_mask_tracing_button = create_text_button(
        "Mask Tracing",
        "Toggle Mask Tracing\n\nSwitch to mask tracing mode to manually draw or edit segmentation masks using brush tools. Toggle again to return to the main view."
    )
    main_window.toggle_mask_tracing_button.setStyleSheet(text_button_style)
    main_window.toggle_mask_tracing_button.clicked.connect(main_window.toggle_mask_tracing)
    correction_layout.addWidget(main_window.toggle_mask_tracing_button)

    correction_section.addLayout(correction_layout)
    correction_widget.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
    )
    top_row_layout.addWidget(correction_widget, 1)

    layout.addLayout(top_row_layout)

    # ========== BOTTOM ROW: CALCULATIONS & VISUALIZATIONS ==========
    bottom_row_layout = QHBoxLayout()
    bottom_row_layout.setSpacing(8)
    bottom_row_layout.setContentsMargins(0, 0, 0, 0)

    # Left column: Calculations section
    calculations_widget = QWidget()
    calculations_widget.setStyleSheet("""
        QWidget {
            background-color: rgba(80, 250, 123, 0.1);
            border: 2px solid #50fa7b;
            border-radius: 6px;
            padding: 6px;
        }
    """)
    calculations_section = QVBoxLayout(calculations_widget)
    calculations_section.setSpacing(4)
    calculations_section.setContentsMargins(0, 0, 0, 0)
    calculations_section.addWidget(create_section_label("Calculations", "#50fa7b"))

    
    calculations_layout = QVBoxLayout()
    calculations_layout.setSpacing(4)
    calculations_layout.setContentsMargins(0, 0, 0, 0)

    # Calculate Root Length button
    main_window.calculate_length_button = create_text_button(
        "Calculate Root Length",
        "Calculate Root Length\n\nMeasure total root length from skeleton data. Processes all skeleton files and calculates cumulative length measurements for each image."
    )
    main_window.calculate_length_button.setStyleSheet(text_button_style)
    main_window.calculate_length_button.clicked.connect(main_window.calculate_root_length)
    calculations_layout.addWidget(main_window.calculate_length_button)

    # Calculate Root Area button
    main_window.calculate_area_button = create_text_button(
        "Calculate Root Area",
        "Calculate Root Area\n\nMeasure root surface area from segmentation masks. Processes all mask files and calculates pixel-based area measurements for each image."
    )
    main_window.calculate_area_button.setStyleSheet(text_button_style)
    main_window.calculate_area_button.clicked.connect(main_window.calculate_root_area)
    calculations_layout.addWidget(main_window.calculate_area_button)

    calculations_section.addLayout(calculations_layout)
    calculations_widget.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
    )
    bottom_row_layout.addWidget(calculations_widget, 1)

    # Right column: Visualizations section
    visualizations_widget = QWidget()
    visualizations_widget.setStyleSheet("""
        QWidget {
            background-color: rgba(255, 121, 198, 0.1);
            border: 2px solid #ff79c6;
            border-radius: 6px;
            padding: 6px;
        }
    """)
    visualizations_section = QVBoxLayout(visualizations_widget)
    visualizations_section.setSpacing(4)
    visualizations_section.setContentsMargins(0, 0, 0, 0)
    visualizations_section.addWidget(create_section_label("Visualizations", "#ff79c6"))
    
    visualizations_layout = QVBoxLayout()
    visualizations_layout.setSpacing(4)
    visualizations_layout.setContentsMargins(0, 0, 0, 0)

    # Visualize Root Length button
    main_window.visualize_root_length_button = create_text_button(
        "Visualize Root Length",
        "Visualize Root Length\n\nOpen interactive dashboard with line charts showing root length analysis. Compare length measurements across different images, tubes, and dates."
    )
    main_window.visualize_root_length_button.setStyleSheet(text_button_style)
    main_window.visualize_root_length_button.clicked.connect(
        main_window.toggle_root_length_visualization
    )
    visualizations_layout.addWidget(main_window.visualize_root_length_button)

    # Visualize Root Area button
    main_window.visualize_root_area_button = create_text_button(
        "Visualize Root Area",
        "Visualize Root Area\n\nOpen interactive dashboard with bar charts showing root area analysis. Compare area measurements across different images, tubes, and dates."
    )
    main_window.visualize_root_area_button.setStyleSheet(text_button_style)
    main_window.visualize_root_area_button.clicked.connect(
        main_window.toggle_root_area_visualization
    )
    visualizations_layout.addWidget(main_window.visualize_root_area_button)

    visualizations_section.addLayout(visualizations_layout)
    visualizations_widget.setSizePolicy(
        QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
    )
    bottom_row_layout.addWidget(visualizations_widget, 1)

    layout.addLayout(bottom_row_layout)

    # ========== SEPARATOR ==========
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setStyleSheet("background-color: #44475a; margin: 8px 0px;")
    separator.setFixedHeight(1)
    layout.addWidget(separator)

    # ========== VIEW MODE COMBO ==========
    main_window.view_mode_combo = QComboBox()
    main_window.view_mode_combo.addItems(["Single Image", "Overlay", "Side by Side"])
    main_window.view_mode_combo.setStyleSheet("""
        QComboBox {
            background-color: #44475a;
            color: #f8f8f2;
            border: 2px solid #6272a4;
            border-radius: 5px;
            padding: 6px 10px;
            font-size: 9pt;
        }
        QComboBox:hover {
            border: 2px solid #8be9fd;
        }
        QComboBox::drop-down {
            border: none;
            padding-right: 8px;
        }
        QComboBox QAbstractItemView {
            background-color: #44475a;
            color: #f8f8f2;
            selection-background-color: #6272a4;
        }
    """)
    main_window.view_mode_combo.currentIndexChanged.connect(
        main_window.display_controller.update_display_mode
    )
    layout.addWidget(main_window.view_mode_combo)

    # ========== TREE CONTROL BUTTONS ==========
    tree_controls_layout = QHBoxLayout()
    tree_controls_layout.setSpacing(5)

    tree_button_style = """
        QPushButton {
            background-color: #44475a;
            color: #f8f8f2;
            border: 1px solid #6272a4;
            border-radius: 4px;
            padding: 5px 10px;
            font-size: 8pt;
        }
        QPushButton:hover {
            background-color: #6272a4;
            border: 1px solid #8be9fd;
        }
    """

    main_window.expand_all_button = QPushButton("➕ Expand All")
    main_window.expand_all_button.setStyleSheet(tree_button_style)
    main_window.expand_all_button.clicked.connect(lambda: main_window.file_list.expandAll())
    main_window.expand_all_button.setMaximumHeight(28)
    tree_controls_layout.addWidget(main_window.expand_all_button)

    main_window.collapse_all_button = QPushButton("➖ Collapse All")
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
