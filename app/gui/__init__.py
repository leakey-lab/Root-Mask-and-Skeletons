"""
GUI Components Package

This package contains all GUI-related components for the Root Viewer application.
Key classes are re-exported here for backward compatibility.
"""

# Main window and core components
from .main_window import MainWindow
from .display_controller import DisplayController, MagnifyingGraphicsView
from .image_manager import ImageManager, ImageLoaderThread

# Mask tracing components
from .mask_tracing_interface import MaskTracingInterface
from .mask_graphics_view import MaskTracingGraphicsView
from .mask_drawing_tools import MaskDrawingMixin
from .mask_cursor_utils import create_brush_cursor, create_panning_cursor

# Image normalization
from .image_normalization_interface import ImageNormalization, NormalizationControls

# UI panel helpers
from . import ui_panels
from . import file_tree_manager
from . import visualization_manager

__all__ = [
    # Main components
    'MainWindow',
    'DisplayController',
    'MagnifyingGraphicsView',
    'ImageManager',
    'ImageLoaderThread',
    # Mask tracing
    'MaskTracingInterface',
    'MaskTracingGraphicsView',
    'MaskDrawingMixin',
    'create_brush_cursor',
    'create_panning_cursor',
    # Image normalization
    'ImageNormalization',
    'NormalizationControls',
    # Modules
    'ui_panels',
    'file_tree_manager',
    'visualization_manager',
]
