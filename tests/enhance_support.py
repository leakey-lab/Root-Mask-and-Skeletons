"""Helpers of the Enhance tests: an EXIF-rotated JPEG, and the repository root.

``exif_rotated_jpeg`` moved here from ``test_enhance_core.py`` (``test_enhance_controls.py``
borrowed it). ``ROOT`` is a copy of ``test_layering_contract.py``'s (that module is frozen in
wave 1; wave 2 can import it from here).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def exif_rotated_jpeg(path, width=80, height=40, orientation=6) -> str:
    """A JPEG stored ``width`` x ``height`` whose EXIF orientation says "rotate on display"."""
    from PIL import Image

    picture = Image.fromarray(np.full((height, width, 3), 128, np.uint8))
    exif = Image.Exif()
    exif[0x0112] = orientation  # 6 = rotate 90 degrees clockwise to display
    picture.save(str(path), exif=exif.tobytes())
    return str(path)
