"""
Visualization Components Package

This package contains all visualization-related components for root length 
and root area analysis. Key classes are re-exported here for backward compatibility.
"""

# Dash application components
from .dash_app import DashApp
from .dash_visualizations import DashVisualizations
from .dash_data_cache import DataCache
from .dash_image_utils import build_available_images_map, get_encoded_image

# Dash area application
from .dash_app_area import DashAppArea

# PyQt visualization windows
from .root_length_visulization import RootLengthVisualization, DashServerThread
from .root_area_visualization import RootAreaVisualization

__all__ = [
    # Dash components
    'DashApp',
    'DashVisualizations',
    'DataCache',
    'build_available_images_map',
    'get_encoded_image',
    'DashAppArea',
    # PyQt windows
    'RootLengthVisualization',
    'RootAreaVisualization',
    'DashServerThread',
]
