"""The render oracles of tests/render_metrics.py measure what they claim: known patterns give
known values, and ``render_view`` grabs items at a known device mapping."""

import cv2
import numpy as np
import pytest

from app.gui.canvas_layers.photo import PhotoLayerItem
from tests.render_metrics import (
    area_reference,
    chirp_alias_std,
    chirp_image,
    equal_neighbour_fraction,
    mean_run,
    psnr,
    remap_reference,
    render_view,
    ring_std,
    sine_image,
    sine_step_ratio,
    stripes_image,
    zone_plate,
)

# Render tests measure the raster engine (offscreen has no GL anyway).
pytestmark = pytest.mark.usefixtures("app", "no_prompts", "gl_off")


# ------------------------------------------------------------------- render metrics
def test_render_metrics_helpers_measure_known_patterns():
    sine = sine_image(64, 8)
    stripes = stripes_image(16, 4)
    assert psnr(sine, sine) == float("inf")
    assert psnr(np.zeros((4, 4)), np.full((4, 4), 255)) == pytest.approx(0.0)
    assert equal_neighbour_fraction(np.zeros((3, 5), np.uint8)) == 1.0
    assert equal_neighbour_fraction(stripes) == 0.0
    assert mean_run(np.repeat(stripes, 4, axis=1)) == pytest.approx(4.0)
    assert sine_step_ratio(sine, sine) == pytest.approx(1.0)
    assert sine_step_ratio(np.repeat(sine, 3, axis=1), sine) == pytest.approx(1.0)  # blocks
    assert stripes.mean() == pytest.approx(127.5)
    assert chirp_alias_std(np.full((8, 200), 127, np.uint8), 0.3) == 0.0
    assert chirp_image().shape == (64, 640)
    assert ring_std(np.full((300, 300), 90, np.uint8), 0.25) == 0.0
    plate = zone_plate(64)
    assert plate.shape == (64, 64) and plate.max() >= 250 and plate.min() <= 5
    src = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(remap_reference(src, 1.0, 0, 0, 4, 3, cv2.INTER_NEAREST), src)
    assert remap_reference(src, 2.0, 0, 0, 8, 6, cv2.INTER_NEAREST)[5, 7] == src[2, 3]
    assert area_reference(np.full((8, 8), 200, np.uint8), 0.5, 0, 0, 4, 4)[1, 1] == 200


def test_render_view_grabs_items_at_a_known_mapping():
    image = np.zeros((40, 60, 3), np.uint8)
    image[:, :, 2] = 255  # BGR red
    item = PhotoLayerItem()
    item.set_image(image)
    render = render_view([item], 2.0, (200, 160))
    assert render.rgb.shape == (160, 200, 3)
    assert render.zoom == pytest.approx(2.0)
    cx, cy = int(render.dx + 60), int(render.dy + 40)  # image centre on the device
    assert tuple(render.rgb[cy, cx]) == (255, 0, 0)
    assert tuple(render.rgb[0, 0]) == (0, 0, 0)  # background outside the 120x80 image
    assert item.scene() is None  # handed back, reusable
