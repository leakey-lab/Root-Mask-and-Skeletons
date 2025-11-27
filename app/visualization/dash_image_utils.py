"""
Image utilities for Dash hover display functionality.
Handles image key management and base64 encoding for thumbnails.
"""

import os
import re
import base64
from PIL import Image
import io


def build_available_images_map(image_manager):
    """
    Build a mapping of (tube, position, date) -> image_path from the image manager.
    
    Args:
        image_manager: ImageManager instance with loaded images
        
    Returns:
        dict: Mapping with tuple keys and path values, plus 'tubes', 'positions', 'dates' sets
    """
    available_images = {}
    
    if image_manager and image_manager.images:
        for name, path in image_manager.images.items():
            tube_match = re.search(r"T(\d+)", name)
            pos_match = re.search(r"L(\d+)", name)
            date_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name)

            if all([tube_match, pos_match, date_match]):
                tube = int(tube_match.group(1))
                position = int(pos_match.group(1))
                date_str = f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}"

                key = (tube, position, date_str)
                available_images[key] = path
                
                if tube not in available_images.get("tubes", set()):
                    available_images.setdefault("tubes", set()).add(tube)
                if position not in available_images.get("positions", set()):
                    available_images.setdefault("positions", set()).add(position)
                if date_str not in available_images.get("dates", set()):
                    available_images.setdefault("dates", set()).add(date_str)
    
    return available_images


def get_encoded_image(available_images, tube: int, position: int, date) -> str:
    """
    Get base64 encoded image for hover display.
    
    Args:
        available_images: Dict mapping (tube, position, date_str) to image paths
        tube: Tube number
        position: Position/depth number
        date: pandas Timestamp or date object
        
    Returns:
        str: Base64 encoded image data URL or empty string if not found
    """
    try:
        date_str = date.strftime("%Y.%m.%d")
        key = (tube, position, date_str)

        if key in available_images:
            path = available_images[key]

            try:
                # Check if file exists and is readable
                if not os.path.exists(path):
                    return ""

                if not os.access(path, os.R_OK):
                    return ""

                with Image.open(path) as img:
                    # Convert to RGB if needed
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGB")

                    img.thumbnail((200, 150), Image.Resampling.LANCZOS)

                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG", optimize=True)

                    img_str = base64.b64encode(buffered.getvalue()).decode()

                    return f"data:image/png;base64,{img_str}"

            except Exception:
                return ""
        else:
            return ""

    except Exception:
        return ""

