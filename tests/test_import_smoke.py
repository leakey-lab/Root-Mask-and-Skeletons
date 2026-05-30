"""
Import-smoke test: every app module must import cleanly.

This is the primary guard for dependency upgrades (dash, PyQt6, numpy, pandas,
plotly, etc.) where the regression suite's pure-function coverage does not reach
GUI / visualization wiring. A broken import after an upgrade fails here.

GUI modules only need a QApplication for *instantiation*, not import, so importing
is safe in a headless/offscreen environment.
"""
import importlib
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Known pre-existing circular import (mask_handler -> mask_tracing_interface ->
# main_window -> mask_handler). Works at runtime only because main.py imports
# main_window first. To be fixed in R5 (decouple handlers from the view); the
# xfail below is removed then.
CIRCULAR_MODULES = {"app.handlers.mask_handler"}

APP_MODULES = [
    "app.mask_model.model",
    "app.inference.skeleton_inference",
    "app.inference.root_length_inference_handler",
    "app.inference.root_area_inference_handler",
    "app.handlers.mask_generation_handler",
    "app.handlers.generate_skeleton_handler",
    "app.handlers.skeleton_handler",
    "app.handlers.mask_handler",
    "app.data_processing.data_processor",
    "app.data_processing.data_processor_area",
    "app.gui.skeleton_graph_model",
    "app.gui.skeleton_correction_graphics_view",
    "app.gui.mask_graphics_view",
    "app.gui.mask_drawing_tools",
    "app.gui.mask_cursor_utils",
    "app.gui.image_manager",
    "app.gui.image_normalization_interface",
    "app.gui.file_tree_manager",
    "app.gui.ui_panels",
    "app.gui.display_controller",
    "app.gui.mask_tracing_interface",
    "app.gui.skeleton_correction_interface",
    "app.gui.visualization_manager",
    "app.gui.main_window",
    "app.visualization.dash_image_utils",
    "app.visualization.dash_data_cache",
    "app.visualization.dash_visualizations",
    "app.visualization.dash_app",
    "app.visualization.dash_app_area",
    "app.visualization.root_length_visulization",
    "app.visualization.root_area_visualization",
]


@pytest.mark.parametrize("modname", APP_MODULES)
def test_module_imports(modname, request):
    if modname in CIRCULAR_MODULES:
        request.applymarker(pytest.mark.xfail(
            reason="pre-existing circular import; fixed in R5", strict=False))
    importlib.import_module(modname)
