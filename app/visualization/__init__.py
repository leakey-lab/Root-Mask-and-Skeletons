"""
Visualization Components Package

This package contains all visualization-related components for root length
and root area analysis. Key classes are re-exported here for backward compatibility.
"""

# Dash application components
from .dash_app import DashApp

# Dash area application
from .dash_app_area import DashAppArea
from .dash_data_cache import DataCache
from .dash_image_utils import build_available_images_map, get_encoded_image
from .dash_visualizations import DashVisualizations
from .root_area_visualization import RootAreaVisualization

# PyQt visualization windows
from .root_length_visulization import DashServerThread, RootLengthVisualization

__all__ = [
    # Dash components
    "DashApp",
    "DashVisualizations",
    "DataCache",
    "build_available_images_map",
    "get_encoded_image",
    "DashAppArea",
    # PyQt windows
    "RootLengthVisualization",
    "RootAreaVisualization",
    "DashServerThread",
]
