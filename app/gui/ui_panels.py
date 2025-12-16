"""
UI Panel creation methods for the main window.
Handles left panel (controls/tree) and right panel (display area) creation.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTreeWidget,
    QLabel,
    QComboBox,
    QStackedWidget,
)
from PyQt6.QtCore import Qt


def create_left_panel(main_window) -> QWidget:
    """
    Create the left panel with action buttons, view mode selector, and file tree.
    
    Args:
        main_window: MainWindow instance to connect signals and store widgets
        
    Returns:
        QWidget: The configured left panel widget
    """
    left_widget = QWidget()
    layout = QVBoxLayout(left_widget)
    layout.setSpacing(8)

    # Style for action buttons
    button_style = """
        QPushButton {
            background-color: #6272a4;
            color: #f8f8f2;
            border: none;
            border-radius: 5px;
            padding: 8px;
            font-weight: bold;
            font-size: 9pt;
        }
        QPushButton:hover {
            background-color: #7282b4;
        }
        QPushButton:pressed {
            background-color: #5262a4;
        }
    """

    main_window.generate_mask_button = QPushButton("🎭 Generate ML Masks")
    main_window.generate_mask_button.setStyleSheet(button_style)
    main_window.generate_mask_button.clicked.connect(
        main_window.mask_generation_handler.generate_masks
    )
    layout.addWidget(main_window.generate_mask_button)

    main_window.generate_button = QPushButton("🦴 Generate Skeleton")
    main_window.generate_button.setStyleSheet(button_style)
    main_window.generate_button.clicked.connect(main_window.skeleton_handler.generate_skeleton)
    layout.addWidget(main_window.generate_button)

    main_window.toggle_skeleton_correction_button = QPushButton("✏️ Skeleton Correction")
    main_window.toggle_skeleton_correction_button.setStyleSheet(button_style)
    main_window.toggle_skeleton_correction_button.clicked.connect(
        main_window.toggle_skeleton_correction
    )
    layout.addWidget(main_window.toggle_skeleton_correction_button)

    main_window.load_images_button = QPushButton("📂 Load Images")
    main_window.load_images_button.setStyleSheet(button_style)
    main_window.load_images_button.clicked.connect(main_window.load_images)
    layout.addWidget(main_window.load_images_button)

    main_window.calculate_length_button = QPushButton("📏 Calculate Root Length")
    main_window.calculate_length_button.setStyleSheet(button_style)
    main_window.calculate_length_button.clicked.connect(main_window.calculate_root_length)
    layout.addWidget(main_window.calculate_length_button)

    main_window.visualize_root_length_button = QPushButton("📊 Visualize Root Length")
    main_window.visualize_root_length_button.setStyleSheet(button_style)
    main_window.visualize_root_length_button.clicked.connect(
        main_window.toggle_root_length_visualization
    )
    layout.addWidget(main_window.visualize_root_length_button)

    main_window.calculate_area_button = QPushButton("📐 Calculate Root Area")
    main_window.calculate_area_button.setStyleSheet(button_style)
    main_window.calculate_area_button.clicked.connect(main_window.calculate_root_area)
    layout.addWidget(main_window.calculate_area_button)

    main_window.visualize_root_area_button = QPushButton("📈 Visualize Root Area")
    main_window.visualize_root_area_button.setStyleSheet(button_style)
    main_window.visualize_root_area_button.clicked.connect(
        main_window.toggle_root_area_visualization
    )
    layout.addWidget(main_window.visualize_root_area_button)

    main_window.toggle_mask_tracing_button = QPushButton("✏️ Toggle Mask Tracing")
    main_window.toggle_mask_tracing_button.setStyleSheet(button_style)
    main_window.toggle_mask_tracing_button.clicked.connect(main_window.toggle_mask_tracing)
    layout.addWidget(main_window.toggle_mask_tracing_button)

    main_window.view_mode_combo = QComboBox()
    main_window.view_mode_combo.addItems(["Single Image", "Overlay", "Side by Side"])
    main_window.view_mode_combo.setStyleSheet("""
        QComboBox {
            background-color: #44475a;
            color: #f8f8f2;
            border: 2px solid #6272a4;
            border-radius: 5px;
            padding: 5px;
            font-size: 9pt;
        }
        QComboBox:hover {
            border: 2px solid #8be9fd;
        }
        QComboBox::drop-down {
            border: none;
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

    # Tree control buttons
    tree_controls_layout = QHBoxLayout()
    tree_controls_layout.setSpacing(5)

    tree_button_style = """
        QPushButton {
            background-color: #44475a;
            color: #f8f8f2;
            border: 1px solid #6272a4;
            border-radius: 3px;
            padding: 5px;
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
    main_window.expand_all_button.setMaximumHeight(30)
    tree_controls_layout.addWidget(main_window.expand_all_button)

    main_window.collapse_all_button = QPushButton("➖ Collapse All")
    main_window.collapse_all_button.setStyleSheet(tree_button_style)
    main_window.collapse_all_button.clicked.connect(lambda: main_window.file_list.collapseAll())
    main_window.collapse_all_button.setMaximumHeight(30)
    tree_controls_layout.addWidget(main_window.collapse_all_button)

    layout.addLayout(tree_controls_layout)

    images_label = QLabel("📂 Image Library:")
    images_label_font = images_label.font()
    images_label_font.setBold(True)
    images_label_font.setPointSize(10)
    images_label.setFont(images_label_font)
    layout.addWidget(images_label)

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

    layout.addWidget(main_window.file_list)

    return left_widget


def create_right_panel(main_window) -> QWidget:
    """
    Create the right panel with the display area.
    
    Args:
        main_window: MainWindow instance to connect display controller
        
    Returns:
        QWidget: The configured right panel widget
    """
    right_widget = QWidget()
    main_window.display_area = main_window.display_controller.setup_display_area(right_widget)
    return right_widget

