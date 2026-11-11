"""The Correct page's view: Enhance, the keyboard focus after clicks on the overlays,
overlays that never collide, the fitted photo clear of them, the zoom readout, the
hand button, and how the photo, the mask and the skeleton are drawn when magnified.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QRect, QRectF
from PyQt6.QtGui import QPainter
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.gui import canvas_gl
from app.gui.correct.layer import CLASSES, MASK, SKELETON
from app.gui.correct.mask_layer import MaskLayer
from app.gui.widgets import (
    tokens,
)
from tests.correct_support import (
    LEFT,
    H,
    Mouse,
    W,
    activate,
    click,
    grab,
    paint,
    plane,
    raster,
    record,
    settle,
    write_image,
    write_skeleton,
)
from tests.qt_teardown import flush_deleted
from tests.render_metrics import sine_image, sine_step_ratio

pytestmark = pytest.mark.usefixtures("correct_guards")


# ================================================================ Enhance, focus, layout
def test_enhancement_renders_live_and_sticks_across_images(make_editor, tmp_path):
    gradient = np.tile(np.linspace(60, 120, W, dtype=np.uint8), (H, 1))
    first, second = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    for path in (first, second):
        cv2.imwrite(path, np.dstack([gradient] * 3))
    editor = make_editor(first)
    raw = editor.photo_item.pixmap().toImage()
    controls = editor.norm_controls

    controls.method_combo.setCurrentText("Contrast Stretching")
    controls.apply_button.click()
    assert controls.is_active() and editor.photo_item.pixmap().toImage() != raw

    editor.load_image(second)
    assert editor.photo_item.pixmap().toImage() != raw  # still enhanced


def test_dock_and_panel_clicks_leave_the_keys_on_the_canvas(make_editor, project):
    editor = make_editor(project.a, project.folder)
    activate(editor)
    saved = record(editor.mask_saved)
    paint(editor, (20, 100), (80, 110))
    view = editor.graphics_view

    QTest.mouseClick(editor.save_button, LEFT)
    assert len(saved) == 1
    targets = [
        editor.controls.size_sliders[MaskLayer.SIZE_SPEC].slider,
        editor.controls.opacity_slider.slider,
        editor.layer_panel.rows[SKELETON],
        editor.layer_panel.rows[MASK].eye_button,
        editor.layer_panel.rows[MASK].eye_button,
        editor.layer_panel.rows[MASK],
    ]
    for widget in targets:
        QTest.mouseClick(widget, LEFT)
        assert QApplication.focusWidget() is view, widget
    editor.enhance_popover.toggle()
    editor.norm_controls.method_combo.setFocus()
    editor.enhance_popover.toggle()
    settle(editor)
    assert QApplication.focusWidget() is view


def test_enhance_applied_again_shows_the_photo_enhanced_once(make_editor, tmp_path):
    """Every render starts from the photo decoded at open: Enhance applied again (Apply
    clicked twice, a slider moved twice) never enhances the enhanced photo."""
    noise = np.random.default_rng(7).integers(60, 120, size=(H, W), dtype=np.uint8)
    path = str(tmp_path / "a.png")
    cv2.imwrite(path, np.dstack([noise] * 3))
    editor = make_editor(path)
    controls = editor.norm_controls
    controls.method_combo.setCurrentText("CLAHE")
    controls.apply_button.click()
    once = editor.photo_item.pixmap().toImage()

    editor.apply_normalization()
    editor.apply_normalization()

    assert editor.photo_item.pixmap().toImage() == once


# Copied unchanged from tests/test_integrator_cross_module.py (A3) at b9f7d19.
def test_editor_enhance_button_shows_whether_the_popover_is_open(app):
    from app.gui.correct.editor import CorrectEditor

    iface = CorrectEditor()
    try:
        assert iface.enhance_button.isCheckable() and not iface.enhance_button.isChecked()
        iface.enhance_button.click()
        assert iface.enhance_button.isChecked() and not iface.enhance_popover.isHidden()
        iface.enhance_popover.toggle()
        assert not iface.enhance_button.isChecked()
    finally:
        iface.close()
        iface.deleteLater()
        flush_deleted()


LONG_NAME = "24EFCAM1_T001_L044_2024.07.18_111530_001_SMM.png"


@pytest.mark.parametrize("layer_id", [MASK, SKELETON, CLASSES])
@pytest.mark.parametrize("size", [(1400, 900), (1000, 700), (820, 560), (700, 520)])
def test_the_overlays_never_collide(make_editor, tmp_path, size, layer_id):
    """The side card (the Layers panel, and the class panel while the class labels are
    edited) is one overlay: its panels are placed inside it."""
    photo = write_image(str(tmp_path / LONG_NAME))
    editor = make_editor(photo, size=size)
    controls = editor.controls
    if layer_id == SKELETON:
        controls.tool_buttons[(SKELETON, "polyline")].click()
        click(editor, (40, 40))
        click(editor, (90, 60))  # the prompt shows
    else:
        controls.tool_buttons[(layer_id, "eraser")].click()
    editor.chrome.enhance_button.click()
    settle(editor)

    chrome = editor.chrome
    candidates = (chrome.rail, chrome.dock, chrome.chip, chrome.zoom, chrome.popover)
    overlays = [w for w in (*candidates, editor.side_panels, controls.prompt) if w.isVisible()]
    assert editor.side_panels in overlays and chrome.popover in overlays
    assert editor.class_panel.isVisible() == (layer_id == CLASSES)
    assert (controls.prompt in overlays) == (layer_id == SKELETON)
    for i, a in enumerate(overlays):
        assert editor.rect().contains(a.geometry()), (a, a.geometry())
        for b in overlays[i + 1 :]:
            assert not a.geometry().intersects(b.geometry()), (a, b, a.geometry(), b.geometry())


def image_rect_in(editor, w: int, h: int) -> QRect:
    """Where the photo is shown, in the editor's coordinates."""
    view = editor.graphics_view
    rect = view.mapFromScene(QRectF(0, 0, w, h)).boundingRect()
    return QRect(view.mapTo(editor, rect.topLeft()), rect.size())


@pytest.mark.parametrize("size", [(907, 562), (1018, 662), (1658, 942)])
def test_fit_keeps_the_photo_clear_of_every_overlay(make_editor, tmp_path, size):
    photo = write_image(str(tmp_path / "plot_001.png"), w=640, h=480)
    editor = make_editor(photo, size=size)
    editor.graphics_view.fit_to_view()
    settle(editor)

    shown = image_rect_in(editor, 640, 480)
    chrome = editor.chrome
    for overlay in (chrome.rail, chrome.dock, chrome.chip, chrome.zoom, editor.side_panels):
        assert not shown.intersects(overlay.geometry()), (overlay, shown, overlay.geometry())
    zoom = editor.graphics_view.zoom_factor
    assert editor.zoom_cluster.zoom_pct_label.text() == f"{round(zoom * 100)}%"


def test_the_zoom_readout_follows_the_canvas_up_to_2000_percent(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view = editor.graphics_view

    view.set_zoom(8.0)
    assert view.zoom_factor == pytest.approx(8.0)
    assert editor.zoom_cluster.zoom_pct_label.text() == "800%"
    view.set_zoom(100.0)
    assert view.zoom_factor == view.MAX_ZOOM


def test_the_hand_button_turns_panning_on_and_off_and_hides_the_ring(make_editor, project):
    editor = make_editor(project.a, project.folder)
    view, pan, ring = editor.graphics_view, editor.chrome.pan_button, editor.layers[SKELETON]
    editor.controls.tool_buttons[(SKELETON, "eraser")].click()
    editor.router.canvas_move(QPoint(100, 60), Mouse(), False)
    assert ring.eraser_ring.is_visible()

    pan.click()
    assert view.pan_mode and not ring.eraser_ring.is_visible()
    pan.click()
    assert not view.pan_mode


# ============================================================================ zoom quality
def sine_photo(tmp_path) -> str:
    path = str(tmp_path / "sine.png")
    cv2.imwrite(path, np.dstack([sine_image(640, 480)] * 3))
    return path


def centred_grab(editor, zoom: float, centre: tuple[float, float]) -> np.ndarray:
    """What the canvas shows at ``zoom`` around image point ``centre`` (no brush ring)."""
    editor.router.canvas_hover_left()
    view = editor.graphics_view
    view.set_zoom(zoom)
    view.centerOn(QPointF(*centre))
    return grab(view)


def content(editor, rgb: np.ndarray, w: int, h: int, margin: int = 3) -> np.ndarray:
    rect = editor.graphics_view.mapFromScene(QRectF(0, 0, w, h)).boundingRect()
    x0, y0 = max(0, rect.left() + margin), max(0, rect.top() + margin)
    x1, y1 = min(rgb.shape[1], rect.right() - margin), min(rgb.shape[0], rect.bottom() - margin)
    return rgb[y0:y1, x0:x1]


@pytest.mark.parametrize("zoom", [4, 8])
def test_mask_pixels_are_exact_blocks_when_magnified(make_editor, tmp_path, zoom):
    photo = write_image(str(tmp_path / "flat.png"), w=640, h=480)
    editor = make_editor(photo)
    editor.controls.size_sliders[MaskLayer.SIZE_SPEC].slider.setValue(2)  # a 1 px dab
    rng = np.random.default_rng(zoom)
    x0, y0, n = 300, 220, 20
    for x, y in rng.integers(0, n, size=(60, 2)):
        click(editor, (x0 + int(x), y0 + int(y)))
    painted_px = plane(editor)[y0 : y0 + n, x0 : x0 + n] > 0
    assert 0 < painted_px.sum() < n * n

    rgb = centred_grab(editor, float(zoom), (x0 + n / 2, y0 + n / 2))
    origin = editor.graphics_view.mapFromScene(QPointF(x0, y0))
    colour = np.array([int(c) for c in bytes.fromhex(tokens.MASK[1:])])
    painted = np.round(0.6 * colour + 0.4 * 90)  # the mask at 60 % over the grey photo
    for row in range(n):
        for col in range(n):
            top, left = origin.y() + row * zoom, origin.x() + col * zoom
            block = rgb[top + 1 : top + zoom - 1, left + 1 : left + zoom - 1].reshape(-1, 3)
            assert (block == block[0]).all(), (row, col)
            expected = painted if painted_px[row, col] else np.array([90, 90, 90])
            assert np.abs(block[0].astype(int) - expected).max() <= 2, (row, col)


def test_the_skeleton_is_a_thin_line_not_blocks_at_8x(make_editor, tmp_path):
    folder = str(tmp_path / "p")
    photo = write_image(os.path.join(folder, "c.png"))
    write_skeleton(os.path.join(folder, "skeletons", "c_fake.png"), row=75)
    editor = make_editor(photo, folder)
    item = editor.layers[SKELETON].item
    item.EXTRACT_BUDGET_MS = float("inf")  # every tile's vectors in the first paint

    rgb = centred_grab(editor, 8.0, (100, 75))

    assert item.mode_painted() in ("vector", "vector-thin")
    column = editor.graphics_view.mapFromScene(QPointF(100.5, 75.5)).x()
    marked = np.abs(rgb[:, column].astype(int) - 90).max(axis=1) > 12
    assert 0 < marked.sum() <= 8  # an 8x block line with its halo would mark 24 rows


def test_an_opengl_swap_keeps_the_canvas_config_and_the_smooth_photo(
    make_editor, tmp_path, monkeypatch
):
    monkeypatch.setattr(canvas_gl, "opengl_viewports_allowed", lambda: True)
    editor = make_editor(show=False)
    view = editor.graphics_view
    assert view.viewport().inherits("QOpenGLWidget")
    hints, background = view.renderHints(), view.backgroundBrush().color().name()

    view.set_opengl_viewport_enabled(False)

    assert not view.viewport().inherits("QOpenGLWidget")
    assert view.renderHints() == hints and not hints & QPainter.RenderHint.SmoothPixmapTransform
    assert view.backgroundBrush().color().name() == background
    assert view.viewport().hasMouseTracking()
    editor.show()
    editor.load_image(sine_photo(tmp_path))
    settle(editor)
    region = content(editor, centred_grab(editor, 4.0, (320, 240)), 640, 480)
    assert sine_step_ratio(region, sine_image(640, 480)) <= 1.5 / 4.0


def test_drawing_the_skeleton_changes_no_skeleton_pixel(make_editor, project):
    editor = make_editor(project.a, project.folder)
    before = raster(editor).copy()

    editor.layers[SKELETON].render_skeleton()
    centred_grab(editor, 8.0, (100, 110))

    assert np.array_equal(raster(editor), before)
