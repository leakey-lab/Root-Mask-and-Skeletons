"""Root area (mm²) from a mask: the root pixels times the area of one pixel of the mask's own
grid, so a mask of any resolution measures the same field of view (F-002).

A mask pixel is root when its grey value is above ``MASK_THRESHOLD`` (127). ``calibration=None``
is the app's field of view (``config.CALIBRATION``, 18 x 13.5 mm). The values here are science
pins: their expected values never change.
"""

import cv2
import numpy as np
import pytest

from app import config
from app.config import Calibration
from app.inference import metrics
from app.inference.metrics import calculate_root_area, calculate_root_length

FOV_W_MM = 18.0
FOV_H_MM = 13.5


@pytest.mark.parametrize("n_px", [1, 100, 200, 1000])
def test_area_scales_with_pixel_count_at_fixed_resolution(n_px):
    w = h = 40
    m = np.zeros((h, w), dtype=np.uint8)
    m.ravel()[:n_px] = 255
    expected = n_px * (FOV_W_MM / w) * (FOV_H_MM / h)
    assert calculate_root_area(m) == pytest.approx(expected, rel=1e-9)


def test_area_resolution_invariant_full_mask():
    # A fully-root mask always measures the whole FOV area, at ANY resolution.
    # This is the core F-002 fix: previously a non-640x480 mask was wrong by the
    # square of the resolution ratio.
    for h, w in [(480, 640), (960, 1280), (240, 320)]:
        m = np.full((h, w), 255, dtype=np.uint8)
        assert calculate_root_area(m) == pytest.approx(FOV_W_MM * FOV_H_MM, rel=1e-9)


def test_the_metric_functions_default_to_the_apps_field_of_view():
    """``calibration=None`` is ``config.CALIBRATION``; the old aliases (still imported from
    outside the app) give the same values."""
    full = np.full((480, 640), 255, np.uint8)
    assert metrics.root_area_mm2(full) == pytest.approx(18.0 * 13.5)
    assert metrics.root_area_mm2(full, calibration=Calibration(18.0, 13.0)) == pytest.approx(234.0)
    assert calculate_root_area(full) == metrics.root_area_mm2(full, calibration=config.CALIBRATION)

    skeleton = np.zeros((480, 640), np.uint8)
    skeleton[240, 100:201] = 255  # 100 orthogonal steps of 18 / 640 mm
    default = metrics.root_length_mm(skeleton)
    assert default == metrics.root_length_mm(skeleton, calibration=config.CALIBRATION)
    assert default == pytest.approx(0.98 * 100 * 18.0 / 640, rel=1e-12)
    assert calculate_root_length(skeleton) == default
    assert metrics.root_length_mm(skeleton, calibration=Calibration(20.0, 13.5)) != default


def test_a_mask_pixel_is_root_above_127(tmp_path):
    """The threshold, at its boundary: 126 and 127 are background, 128 and 255 are root."""
    assert metrics.MASK_THRESHOLD == 127
    mask = np.zeros((40, 40), np.uint8)
    flat = mask.ravel()
    flat[0:3] = 126
    flat[10:15] = 127
    flat[20:27] = 128
    flat[40:51] = 255
    root_pixels = 7 + 11  # the 128s and the 255s
    pixel_mm2 = (18.0 / 40) * (13.5 / 40)
    path = tmp_path / "x_T1_L2_2024.01.02_010200_001.png"
    assert cv2.imwrite(str(path), mask)

    assert int(np.count_nonzero(metrics.load_mask(str(path)) == 255)) == root_pixels
    assert metrics.root_area_mm2(metrics.load_mask(str(path))) == pytest.approx(
        root_pixels * pixel_mm2, rel=1e-12
    )
    row = metrics.area_row(path.stem, str(path))
    assert row[metrics.AREA_VALUE_COLUMN] == round(root_pixels * pixel_mm2, 2) == 2.73
    assert row["_non_binary"] is True
