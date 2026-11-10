"""The GT-first project of the Measure tests: photos, predictions and GT files whose stems
each pin one rule of ``app.inference.measure_sources``.

Moved out of ``test_measure_sources.py`` so that ``test_measure_gt_first.py`` need not
import a test module. Its ``_write_png`` writes through ``open`` (folders made as needed),
unlike the one of ``tests/measure_support.py``.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

from app.config import SKELETON_OUTPUT_SIZE
from app.editing.thinning import thin_raster
from app.masks.store import write_mask_file
from app.project_layout import ProjectLayout

PHOTO_SIZE = (640, 480)  # (w, h)
#: p prediction only, g prediction + valid GT, o GT only, s GT skeleton of the prediction's
#: size, u GT skeleton equal to the prediction as opened, x undecodable GT, m GT mask equal
#: to the prediction mask.
STEMS = ("p", "g", "o", "s", "u", "x", "m")
NOT_A_PNG = b"this is not an image"


def _write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _write_png(path: str, gray: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", gray)
    assert ok
    _write(path, buf.tobytes())


def write_photo(layout: ProjectLayout, stem: str, size: tuple[int, int] = PHOTO_SIZE) -> str:
    """A JPEG photo ``<root>/<stem>.jpg`` of ``size`` ``(w, h)`` (640 x 480)."""
    w, h = size
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.circle(img, (w // 2, h // 2), 100, (160, 170, 180), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    path = os.path.join(layout.root, stem + ".jpg")
    _write(path, buf.tobytes())
    return path


def prediction_skeleton() -> np.ndarray:
    """A generated skeleton: a soft 3 px line on black at ``SKELETON_OUTPUT_SIZE``."""
    w, h = SKELETON_OUTPUT_SIZE
    gray = np.zeros((h, w), np.uint8)
    cv2.line(gray, (20, 30), (w - 40, h - 30), 255, 3)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def gt_skeleton(size: tuple[int, int] = PHOTO_SIZE) -> np.ndarray:
    """A hand-traced 1 px skeleton at ``size`` ``(w, h)``, unlike the prediction."""
    w, h = size
    gray = np.zeros((h, w), np.uint8)
    cv2.line(gray, (w // 10, h - h // 8), (w - w // 12, h // 8), 255, 1)
    return gray


def prediction_mask() -> np.ndarray:
    w, h = PHOTO_SIZE
    gray = np.zeros((h, w), np.uint8)
    cv2.rectangle(gray, (100, 120), (300, 200), 255, -1)
    return gray


def write_gt_skeleton(layout: ProjectLayout, stem: str, gray: np.ndarray | None = None) -> str:
    path = layout.gt_skeleton_path(stem)
    _write_png(path, gt_skeleton() if gray is None else gray)
    return path


def write_gt_mask(layout: ProjectLayout, stem: str, on: np.ndarray | None = None) -> str:
    """A GT mask as Correct saves it: white on transparent RGBA at the photo's size."""
    if on is None:
        on = np.zeros(PHOTO_SIZE[::-1], bool)
        on[300:400, 50:250] = True
    path = layout.gt_mask_path(stem)
    write_mask_file(path, on, durable=False)
    return path


def make_gt_project(root: str) -> tuple[ProjectLayout, dict[str, str]]:
    """The project of :data:`STEMS`: ``(layout, images)``."""
    layout = ProjectLayout(root)
    images = {stem: write_photo(layout, stem) for stem in STEMS}
    skeleton = prediction_skeleton()
    mask = prediction_mask()
    for stem in STEMS:
        if stem == "o":
            continue
        _write_png(layout.skeleton_path(stem), skeleton)
        _write_png(layout.mask_path(stem), mask)
    write_gt_skeleton(layout, "g")
    write_gt_skeleton(layout, "o")
    write_gt_skeleton(layout, "s", gt_skeleton(SKELETON_OUTPUT_SIZE))
    write_gt_skeleton(layout, "u", thin_raster(skeleton, PHOTO_SIZE))
    _write(layout.gt_skeleton_path("x"), NOT_A_PNG)
    write_gt_mask(layout, "o")
    write_gt_mask(layout, "m", mask > 127)
    _write(layout.gt_mask_path("x"), NOT_A_PNG)
    return layout, images
