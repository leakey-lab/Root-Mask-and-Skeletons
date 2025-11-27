"""
Utility for resolving resource paths in both development and PyInstaller environments
"""

import sys
import os


def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.

    Args:
        relative_path: Path relative to the project root

    Returns:
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # In development, go up from resources/ to project root
        # This file is in resources/, so go up one level to get project root
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, relative_path)


def get_app_dir():
    """
    Get the application directory (where the executable or main script is located).

    Returns:
        Absolute path to application directory
    """
    try:
        # PyInstaller extracts files to _MEIPASS temp folder
        return sys._MEIPASS
    except Exception:
        # In development, return the directory containing this module
        return os.path.dirname(os.path.abspath(__file__))
