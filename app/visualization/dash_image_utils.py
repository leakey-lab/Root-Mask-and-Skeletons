"""
Image utilities for Dash hover display functionality.
Handles image key management and base64 encoding for thumbnails.

Encoded thumbnails are cached on the map built by
:func:`build_available_images_map` and encoded lazily on first request by
:func:`get_encoded_image`, then memoized. A hovered key is read/decoded/
resized/encoded from disk at most once; repeat hovers are O(1). Encoding is
NOT done eagerly at build time — that scaled O(all loaded images) and blocked
the visualization open for minutes on large datasets (F-013).
"""

import os
import re
import base64
import logging
from PIL import Image, UnidentifiedImageError
import io

logger = logging.getLogger(__name__)

# Internal key under which the precomputed base64 thumbnail cache is stored on
# the available-images map. Using a string key keeps it disjoint from the
# (tube, position, date_str) tuple path keys.
_THUMBNAIL_CACHE_KEY = "_encoded_thumbnails"

# Thumbnail dimensions for hover previews. Sized so each image renders as a
# meaningful >=100x100 representation in the Growth-Lines side panel (~300px
# wide) without visible upscaling blur.
_THUMBNAIL_SIZE = (256, 256)


def _encode_thumbnail(path: str) -> str:
    """
    Read, decode, resize and base64-encode a single image as a PNG data URL.

    Returns an empty string if the file is missing, unreadable, or not a valid
    image. Raised exceptions are limited to the specific I/O / decode errors we
    expect from the filesystem and PIL.
    """
    try:
        if not os.path.exists(path) or not os.access(path, os.R_OK):
            return ""

        with Image.open(path) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            img.thumbnail(_THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            img.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.warning("Failed to encode thumbnail for %s: %s", path, exc)
        return ""


def build_available_images_map(image_manager) -> dict:
    """
    Build a mapping of (tube, position, date) -> image_path from the image
    manager. Thumbnails are NOT encoded here; the returned map carries an empty
    cache that :func:`get_encoded_image` fills lazily on first hover.

    Args:
        image_manager: ImageManager instance with loaded images

    Returns:
        dict: Mapping with (tube, position, date_str) tuple keys -> path values.
            A private "_encoded_thumbnails" entry holds the
            {(tube, position, date_str): data_url} cache, populated lazily by
            :func:`get_encoded_image`.
    """
    available_images: dict = {}
    thumbnails: dict = {}

    if image_manager and image_manager.images:
        for name, path in image_manager.images.items():
            tube_match = re.search(r"T(\d+)", name)
            pos_match = re.search(r"L(\d+)", name)
            date_match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", name)

            if all([tube_match, pos_match, date_match]):
                tube = int(tube_match.group(1))
                position = int(pos_match.group(1))
                date_str = (
                    f"{date_match.group(1)}.{date_match.group(2)}.{date_match.group(3)}"
                )

                key = (tube, position, date_str)
                available_images[key] = path
                # F-013: thumbnails are encoded lazily on first hover and
                # memoized by get_encoded_image(), NOT eagerly here. Eager
                # precompute scaled O(all loaded images): 46k images blocked
                # the open for ~6 min. Lazy keeps open fast; hover pays the
                # ~8ms encode once per (tube, position, date) key.

    available_images[_THUMBNAIL_CACHE_KEY] = thumbnails
    return available_images


def get_encoded_image(available_images, tube: int, position: int, date) -> str:
    """
    Get base64 encoded image for hover display from the precomputed cache.

    Args:
        available_images: Map returned by :func:`build_available_images_map`.
        tube: Tube number
        position: Position/depth number
        date: pandas Timestamp or date object

    Returns:
        str: Base64 encoded image data URL or empty string if not found.
    """
    try:
        date_str = date.strftime("%Y.%m.%d")
    except (AttributeError, ValueError) as exc:
        logger.debug("Invalid date passed to get_encoded_image: %s (%s)", date, exc)
        return ""

    key = (tube, position, date_str)

    thumbnails = available_images.get(_THUMBNAIL_CACHE_KEY)
    if isinstance(thumbnails, dict):
        # Fast path: precomputed thumbnail (F-013).
        if key in thumbnails:
            return thumbnails[key]
        # No image known for this key.
        if key not in available_images:
            return ""
        # Known path but missing from cache (e.g. map built externally):
        # encode lazily and memoize.
        encoded = _encode_thumbnail(available_images[key])
        thumbnails[key] = encoded
        return encoded

    # Fallback for maps without a precomputed cache.
    if key in available_images:
        return _encode_thumbnail(available_images[key])
    return ""
